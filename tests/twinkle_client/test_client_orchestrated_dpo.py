import asyncio

from cookbook.client.async_rl.client_orchestrated_dpo import (
    prepare_dpo_batch,
    run_dpo,
)
from twinkle_client.types import DataRef


def test_prepare_dpo_batch_interleaves_complete_pairs() -> None:
    batch = [
        {
            'pair_id': 'a',
            'positive': {'input_ids': [1, 2], 'labels': [-100, 2]},
            'negative': {'input_ids': [1, 3], 'labels': [-100, 3]},
        },
        {
            'pair_id': 'b',
            'positive': {'input_ids': [4], 'labels': [4]},
            'negative': {'input_ids': [5], 'labels': [5]},
        },
    ]

    rows = prepare_dpo_batch(batch)

    assert [row['pair_id'] for row in rows] == ['a', 'a', 'b', 'b']
    assert [row['input_ids'] for row in rows] == [[1, 2], [1, 3], [4], [5]]


def test_dpo_roles_overlap_reference_and_training(monkeypatch) -> None:
    import cookbook.client.async_rl.client_orchestrated_dpo as module

    monkeypatch.setattr(module, 'MAX_STEPS', 2)
    first_train_started = asyncio.Event()
    events = []

    class FakeDataPlane:

        def __init__(self):
            self.rows = {}
            self.released = []

        async def aput(self, rows, *, kind, tags=None):
            ref = DataRef(
                ref_id=f'{kind}-{len(self.rows)}',
                size=len(rows),
                fields=list(rows[0]),
                kind=kind,
            )
            self.rows[ref.ref_id] = rows
            return ref

        async def aappend(self, ref, updates, *, tags=None):
            self.rows[ref.ref_id] = [
                {**row, **update}
                for row, update in zip(self.rows[ref.ref_id], updates)
            ]
            return ref.model_copy(update={'fields': list(self.rows[ref.ref_id][0])})

        async def arelease(self, ref):
            self.released.append(ref.ref_id)

    class FakeModel:

        def __init__(self):
            self.references = 0
            self.steps = 0
            self.forward_backward_kwargs = []

        async def forward_only(self, ref, **kwargs):
            self.references += 1
            name = f'reference-{self.references}'
            events.append(f'{name}-start')
            if self.references == 2:
                await first_train_started.wait()
            events.append(f'{name}-done')
            assert kwargs['output_ref'] == ref
            assert kwargs['output_fields'] == {'logps': 'ref_logps'}
            return ref.model_copy(update={'fields': [*ref.fields, 'ref_logps']})

        async def forward_backward(self, _ref, **kwargs):
            self.forward_backward_kwargs.append(kwargs)
            events.append('train-start')
            first_train_started.set()

        async def clip_grad_and_step(self, **_kwargs):
            self.steps += 1

        async def save(self, name, **_kwargs):
            return {'twinkle_path': name}

    batches = [
        [{'pair_id': 'a', 'positive': {'input_ids': [1]}, 'negative': {'input_ids': [2]}}],
        [{'pair_id': 'b', 'positive': {'input_ids': [3]}, 'negative': {'input_ids': [4]}}],
    ]
    model = FakeModel()
    data_plane = FakeDataPlane()

    saved = asyncio.run(run_dpo(batches, model, data_plane))

    assert events.index('train-start') < events.index('reference-2-done')
    assert model.steps == 2
    assert model.forward_backward_kwargs == [
        {'kwarg_fields': {'ref_outputs.logps': 'ref_logps'}},
        {'kwarg_fields': {'ref_outputs.logps': 'ref_logps'}},
    ]
    assert saved == {'twinkle_path': 'dpo-policy-2'}
    assert len(data_plane.released) == 2
