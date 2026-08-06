from typing import Any, Dict, List, Optional, Union
from twinkle_client.http import http_post
from twinkle_client.types.sampler import AddAdapterResponse, SampleResponseModel, SetTemplateResponse
from peft import PeftConfig
from twinkle.data_format import Trajectory, InputFeature
from twinkle_client.common.json_utils import json_safe
from twinkle_client.remote_task import RemoteTask
from twinkle_client.types.component import ComponentTaskRef, DataRef


# Intentionally does NOT subclass ``twinkle.sampler.base.Sampler``: importing
# that base pulls ``twinkle.sampler.__init__`` → ``VLLMEngine`` → torch + zmq,
# which the mock / CPU-only client environments don't have.
def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy arrays / torch tensors to JSON-serialisable lists.

    ``sample()`` accepts pre-encoded ``InputFeature`` dicts (e.g. from a multi-turn
    rollout's ``template.encode``) whose values are numpy arrays or torch tensors;
    these are not JSON-serialisable and would break the HTTP POST. Detection is by
    duck-typing (``.tolist()``) so this stays free of a hard torch/numpy import,
    honouring the CPU-only client contract noted above.
    """
    return json_safe(obj)


class vLLMSampler:
    """Client wrapper for Sampler that calls server HTTP endpoints.

    This client manages sampling operations and adapter synchronization with the sampler server.
    The server-side session (managed by TwinkleClient) keeps the sampler alive.
    """

    def __init__(self, model_id: str, **kwargs):
        """Create the sampler instance on server."""
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()
        from twinkle_client.data_plane import DataPlaneClient
        self.data_plane = DataPlaneClient(kwargs.pop('data_plane_url', None))

        self.adapter_name = None
        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.model_id = model_id
        self.server_url = f'{self.server_url}/sampler/{model_id}/twinkle'
        response = http_post(
            url=f'{self.server_url}/create',
            json_data=kwargs
        )
        response.raise_for_status()

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig, **kwargs) -> AddAdapterResponse:
        """Add a new adapter to the sampler."""
        if isinstance(config, PeftConfig):
            config = config.__dict__
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_sampler',
            json_data={'adapter_name': adapter_name, 'config': config, **kwargs}
        )
        response.raise_for_status()
        self.adapter_name = adapter_name
        return AddAdapterResponse(**response.json())

    def sample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature]],
        sampling_params: Optional[Dict[str, Any]] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        num_samples: int = 1,
    ) -> List[SampleResponseModel]:
        """Sample from the model.

        Args:
            inputs: List of Trajectory or InputFeature to sample from.
            sampling_params: Sampling parameters dict.
            adapter_name: Adapter name for LoRA inference.
            adapter_uri: Adapter URI (twinkle:// path or local path) for LoRA inference.
            num_samples: Number of completions to generate per prompt.

        Returns:
            SampleResponseModel with 'sequences' list, each containing tokens, logprobs, stop_reason.
        """
        sampling_params = dict(sampling_params or {})
        sampling_params['num_samples'] = num_samples
        json_data = {
            'inputs': _json_safe(inputs),
            'sampling_params': sampling_params,
            'adapter_name': adapter_name,
            'num_samples': num_samples,
        }
        if adapter_uri is not None:
            json_data['adapter_uri'] = adapter_uri

        response = http_post(
            url=f'{self.server_url}/sample',
            json_data=json_data
        )
        response.raise_for_status()
        return [SampleResponseModel(**r) for r in response.json()['samples']]

    def submit_sample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> RemoteTask:
        """Submit directly to the Sampler component and return immediately."""
        body = {
            'sampling_params': sampling_params,
            'adapter_name': adapter_name,
            'adapter_uri': adapter_uri,
            'policy_version': policy_version,
            'group_ids': group_ids,
            'num_samples': num_samples,
        }
        body['input_ref' if isinstance(inputs, DataRef) else 'inputs'] = (
            inputs.model_dump() if isinstance(inputs, DataRef) else _json_safe(inputs))
        response = http_post(
            url=f'{self.server_url}/submit_sample',
            json_data=json_safe(body),
        )
        response.raise_for_status()
        return RemoteTask(ComponentTaskRef(**response.json()))

    async def asample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> List[SampleResponseModel]:
        """Submit sampling and asynchronously await it without blocking the event loop."""
        import asyncio
        task = await asyncio.to_thread(
            self.submit_sample,
            inputs,
            sampling_params,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            policy_version=policy_version,
            group_ids=group_ids,
            num_samples=num_samples,
        )
        result = await task.aresult()
        if isinstance(result, dict) and result.get('output_ref'):
            output_ref = DataRef(**result['output_ref'])
            try:
                batch = await self.data_plane.aget_batch(output_ref)
            finally:
                await self.data_plane.arelease(output_ref)
            if batch.tags and all('prompt_index' in tag for tag in batch.tags):
                grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
                for row, tag in zip(batch.rows, batch.tags):
                    grouped.setdefault(int(tag['prompt_index']), []).append(
                        (int(tag.get('generation_idx', 0)), row))
                samples = []
                for prompt_index in sorted(grouped):
                    generation_rows = [row for _, row in sorted(grouped[prompt_index])]
                    first = generation_rows[0]
                    samples.append({
                        'sequences': [{
                            key: value
                            for key, value in row.items()
                            if key not in ('prompt_logprobs', 'topk_prompt_logprobs')
                        } for row in generation_rows],
                        'prompt_logprobs': first.get('prompt_logprobs'),
                        'topk_prompt_logprobs': first.get('topk_prompt_logprobs'),
                    })
            else:
                # Compatibility with a server that still stores one nested row per prompt.
                samples = batch.rows
        else:
            samples = result.get('samples', []) if isinstance(result, dict) else []
        return [SampleResponseModel(**item) for item in samples]

    def unload_adapter_paths(self, adapter_paths: list[str]) -> None:
        """Evict policy snapshots that are no longer referenced by this client."""
        response = http_post(
            url=f'{self.server_url}/unload_adapter_paths',
            json_data={'adapter_paths': adapter_paths},
        )
        response.raise_for_status()

    def set_template(self, template_cls: str, adapter_name: str = '', **kwargs) -> SetTemplateResponse:
        """Set the template for encoding trajectories."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={'template_cls': template_cls, 'adapter_name': adapter_name, **kwargs}
        )
        response.raise_for_status()
        return SetTemplateResponse(**response.json())
    
    def apply_patch(self, patch_cls: str, **kwargs) -> None:
        """Apply a patch to the model."""
        response = http_post(
            url=f'{self.server_url}/apply_patch',
            json_data={'patch_cls': patch_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
