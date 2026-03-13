# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
from unittest import mock

import torch

from twinkle.model.transformers.strategy.sequence_parallel import (
    DistributedAttention,
    _assert_attention_mask_matches_sequence,
    _gather_attention_mask_for_sp,
)


class TestAttentionMaskSemantics(unittest.TestCase):

    @staticmethod
    def _fake_all_gather_with_rank_offset(output: torch.Tensor, input_tensor: torch.Tensor, group=None):
        del group
        # Simulate two SP ranks with deterministic distinct chunks.
        output.copy_(torch.cat([input_tensor, input_tensor + 10], dim=0))

    @staticmethod
    def _fake_all_gather_duplicate(output: torch.Tensor, input_tensor: torch.Tensor, group=None):
        del group
        output.copy_(torch.cat([input_tensor, input_tensor], dim=0))

    def test_gather_attention_mask_2d(self):
        local_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=torch.int64,
        )
        with mock.patch('torch.distributed.all_gather_into_tensor', self._fake_all_gather_with_rank_offset):
            gathered = _gather_attention_mask_for_sp(
                local_mask,
                local_seq_len=3,
                sp_world_size=2,
                sp_group=None,
            )
        expected = torch.cat([local_mask, local_mask + 10], dim=1)
        self.assertTrue(torch.equal(gathered, expected))
        _assert_attention_mask_matches_sequence(gathered, expected_seq_len=6)

    def test_gather_attention_mask_4d_splits_both_sequence_dims(self):
        local_mask = torch.arange(2 * 1 * 3 * 3, dtype=torch.float32).view(2, 1, 3, 3)
        with mock.patch('torch.distributed.all_gather_into_tensor', self._fake_all_gather_duplicate):
            gathered = _gather_attention_mask_for_sp(
                local_mask,
                local_seq_len=3,
                sp_world_size=2,
                sp_group=None,
            )
        expected = torch.cat(
            [
                torch.cat([local_mask, local_mask], dim=-1),
                torch.cat([local_mask, local_mask], dim=-1),
            ],
            dim=-2,
        )
        self.assertEqual(gathered.shape, (2, 1, 6, 6))
        self.assertTrue(torch.equal(gathered, expected))
        _assert_attention_mask_matches_sequence(gathered, expected_seq_len=6)

    def test_assert_attention_mask_mismatch_raises(self):
        bad_mask = torch.ones((2, 3), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, 'incompatible with attention sequence length'):
            _assert_attention_mask_matches_sequence(bad_mask, expected_seq_len=6)

    def test_distributed_attention_decode_mask_uses_kv_length(self):
        captured = {}

        def local_attn(query, key, value, attention_mask, *args, **kwargs):
            del key, value, args, kwargs
            captured['attention_mask_shape'] = tuple(attention_mask.shape)
            return query

        sequence_parallel = mock.Mock(world_size=2, sp_world_size=2, _sp_group=None)
        attention = DistributedAttention(local_attn, sequence_parallel)
        query = torch.randn(2, 1, 4, 8)
        key = torch.randn(2, 3, 4, 8)
        value = torch.randn(2, 3, 4, 8)
        local_mask = torch.ones((2, 3), dtype=torch.int64)

        with mock.patch('torch.distributed.all_gather_into_tensor', self._fake_all_gather_duplicate), mock.patch(
                'twinkle.model.transformers.strategy.sequence_parallel._SeqAllToAll.apply',
                side_effect=lambda group, tensor, scatter_idx, gather_idx: tensor):
            output = attention(query, key, value, local_mask)

        self.assertEqual(output.shape, query.shape)
        self.assertEqual(captured['attention_mask_shape'], (2, 6))


if __name__ == '__main__':
    unittest.main()
