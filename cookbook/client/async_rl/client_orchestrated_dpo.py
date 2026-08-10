"""Runnable offline DPO composed from the low-level async component clients."""
from __future__ import annotations

import asyncio
import inspect
import os
from typing import Any

from peft import LoraConfig

from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.preprocessor import EmojiDPOProcessor
from twinkle_client import DataPlaneClient, init_twinkle_client
from twinkle_client.async_rl import Worker, WorkerPipeline
from twinkle_client.common.serialize import json_safe
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.types import DataRef

BASE_MODEL = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen3.5-4B')
MODEL_ID = f'ms://{BASE_MODEL}'
TEMPLATE_CLS = os.environ.get(
    'TWINKLE_TEMPLATE_CLS',
    'Qwen3_5Template' if ('Qwen3.5' in BASE_MODEL or 'Qwen3.6' in BASE_MODEL) else 'Template',
)
DATASET_ID = os.environ.get('TWINKLE_DPO_DATASET_ID', 'ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji')
ADAPTER_NAME = os.environ.get('TWINKLE_ADAPTER_NAME', 'client-dpo')
MAX_STEPS = int(os.environ.get('TWINKLE_MAX_STEPS', '100'))
BATCH_SIZE = int(os.environ.get('TWINKLE_BATCH_SIZE', '4'))
MAX_LENGTH = int(os.environ.get('TWINKLE_MAX_LENGTH', '2048'))


def create_dataset() -> Dataset:
    """Load and encode preference pairs in the client process."""
    dataset = Dataset(DatasetMeta(DATASET_ID, data_slice=range(MAX_STEPS * BATCH_SIZE)))
    dataset.set_template(TEMPLATE_CLS, model_id=MODEL_ID, max_length=MAX_LENGTH)
    dataset.map(EmojiDPOProcessor, init_args={'system': 'You are a helpful assistant.'})
    dataset.encode()
    return dataset


def prepare_dpo_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten pairs as ``chosen_0, rejected_0, ...`` for DP-safe slicing."""
    rows: list[dict[str, Any]] = []
    for pair in batch:
        common = {key: value for key, value in pair.items() if key not in ('positive', 'negative')}
        rows.append({**common, **pair['positive']})
        rows.append({**common, **pair['negative']})
    return json_safe(rows)


def _extract_ref_outputs(result: Any, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Normalize an async forward-only result into the DPOLoss input shape."""
    payload: Any = rows
    if payload is None and isinstance(result, dict):
        payload = result.get('result', result)
    if isinstance(payload, list) and payload and all(
            isinstance(row, dict) and row.get('logps') is not None for row in payload):
        return {'logps': [row['logps'] for row in payload]}
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0].get('result', payload[0])
    if not isinstance(payload, dict) or payload.get('logps') is None:
        raise RuntimeError('reference forward did not return per-token logps')
    return {'logps': payload['logps']}


async def _put_rows(data_plane, rows, *, kind, tags):
    try:
        return await data_plane.aput(rows, kind=kind, tags=tags)
    except TypeError as error:
        if 'tags' not in str(error):
            raise
        return await data_plane.aput(rows, kind=kind)


async def _submit(method, *args, **kwargs):
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)
    task = await asyncio.to_thread(method, *args, **kwargs)
    if inspect.isawaitable(task):
        return await task
    return task


class _DatasetWorker(Worker):

    def __init__(self, dataloader, data_plane, output):
        super().__init__('dataset')
        self.dataloader = dataloader
        self.data_plane = data_plane
        self.output = output

    async def run(self) -> None:
        completed = 0
        for batch in self.dataloader:
            if completed >= MAX_STEPS:
                break
            rows = prepare_dpo_batch(batch)
            tags = []
            for index in range(0, len(rows), 2):
                source_pair_id = rows[index].get('pair_id', f'pair-{completed}-{index // 2}')
                tags.extend((
                    {
                        'record_type': 'preference',
                        'pair_id': str(source_pair_id),
                        'pair_role': 'chosen',
                        'pair_status': 'DATA_READY',
                    },
                    {
                        'record_type': 'preference',
                        'pair_id': str(source_pair_id),
                        'pair_role': 'rejected',
                        'pair_status': 'DATA_READY',
                    },
                ))
            ref = await _put_rows(
                self.data_plane, rows, kind='dpo-preference', tags=tags)
            await self.output.put(ref)
            completed += 1
        await self.output.put(None)


class _ReferenceWorker(Worker):

    def __init__(self, model, data_plane, source, output):
        super().__init__('reference')
        self.model = model
        self.data_plane = data_plane
        self.source = source
        self.output = output

    async def run(self) -> None:
        while True:
            item = await self.source.get()
            if item is None:
                await self.output.put(None)
                return
            ref = item
            ref_outputs = await _reference_forward(self.model, self.data_plane, ref)
            await self.output.put((ref, ref_outputs))


class _TrainerWorker(Worker):

    def __init__(self, model, data_plane, source):
        super().__init__('trainer')
        self.model = model
        self.data_plane = data_plane
        self.source = source
        self.completed_steps = 0
        self.saved = None

    async def run(self) -> None:
        while True:
            item = await self.source.get()
            if item is None:
                self.saved = await _submit(
                    self.model.submit_save,
                    f'dpo-policy-{self.completed_steps}',
                    save_optimizer=True,
                )
                return
            ref, ref_outputs = item
            try:
                await _submit(
                    self.model.submit_forward_backward,
                    ref,
                    ref_outputs=ref_outputs,
                )
                await _submit(self.model.submit_clip_grad_and_step, max_grad_norm=1.0)
            finally:
                await self.data_plane.arelease(ref)
            self.completed_steps += 1


async def _reference_forward(
    model: MultiLoraTransformersModel,
    data_plane: DataPlaneClient,
    batch_ref: DataRef,
) -> dict[str, Any]:
    """Run the frozen base model and materialize its DataPlane result."""
    result = await _submit(model.submit_forward_only, batch_ref, disable_lora=True)
    if not isinstance(result, dict) or not result.get('output_ref'):
        return _extract_ref_outputs(result)

    output_ref = DataRef(**result['output_ref'])
    try:
        rows = await data_plane.aget(output_ref)
        return _extract_ref_outputs(result, rows)
    finally:
        await data_plane.arelease(output_ref)


async def run_dpo(
    dataloader: DataLoader,
    model: MultiLoraTransformersModel,
    data_plane: DataPlaneClient,
) -> dict[str, Any]:
    """Run client-owned DPO roles over the shared Model and DataPlane services."""
    preference_ready = asyncio.Queue(maxsize=2)
    reference_ready = asyncio.Queue(maxsize=2)
    trainer = _TrainerWorker(model, data_plane, reference_ready)
    await WorkerPipeline((
        _DatasetWorker(dataloader, data_plane, preference_ready),
        _ReferenceWorker(model, data_plane, preference_ready, reference_ready),
        trainer,
    )).run()
    return trainer.saved


async def train() -> None:
    client = init_twinkle_client(
        base_url=os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000'),
        api_key=os.environ.get('TWINKLE_SERVER_TOKEN', 'EMPTY_TOKEN'),
    )
    model = MultiLoraTransformersModel(MODEL_ID)
    data_plane = DataPlaneClient()

    model.add_adapter_to_model(
        ADAPTER_NAME,
        LoraConfig(target_modules='all-linear', r=8, lora_alpha=32, lora_dropout=0.05),
    )
    model.set_template(TEMPLATE_CLS, model_id=MODEL_ID)
    model.set_processor('InputProcessor', padding_side='right')
    model.set_loss('DPOLoss', beta=0.1, loss_type='sigmoid', reference_free=False)
    model.add_metric('DPOMetric', beta=0.1)
    model.set_optimizer('AdamW', lr=1e-5)

    try:
        dataloader = DataLoader(dataset=create_dataset(), batch_size=BATCH_SIZE, num_workers=0)
        saved = await run_dpo(dataloader, model, data_plane)
        print(f"saved DPO adapter to {saved['twinkle_path']}")
    finally:
        client.close()


if __name__ == '__main__':
    asyncio.run(train())
