# Copyright (c) ModelScope Contributors. All rights reserved.
import unittest
import torch

from twinkle.model.transformers.strategy.sequence_parallel import (SequenceParallel, _extract_text_position_ids,
                                                                   get_cu_seqlens_from_position_ids)


class TestSequenceParallelPositionIdsCompat(unittest.TestCase):

    def test_extract_text_position_ids_for_1d_2d_3d_4d(self):
        pos_1d = torch.arange(6)
        out_1d = _extract_text_position_ids(pos_1d)
        self.assertEqual(out_1d.shape, (1, 6))
        self.assertTrue(torch.equal(out_1d[0], pos_1d))

        pos_2d = torch.arange(12).view(2, 6)
        out_2d = _extract_text_position_ids(pos_2d)
        self.assertTrue(torch.equal(out_2d, pos_2d))

        pos_3d = torch.arange(3 * 2 * 6).view(3, 2, 6)
        out_3d = _extract_text_position_ids(pos_3d)
        self.assertTrue(torch.equal(out_3d, pos_3d[0]))

        pos_4d = torch.arange(4 * 2 * 6).view(4, 2, 6)
        out_4d = _extract_text_position_ids(pos_4d)
        self.assertTrue(torch.equal(out_4d, pos_4d[0]))

    def test_get_cu_seqlens_supports_2d_and_4d(self):
        pos_2d = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
        pos_4d = torch.stack([pos_2d, pos_2d + 10, pos_2d + 20, pos_2d + 30], dim=0)
        expected = torch.tensor([0, 3, 6], dtype=torch.long)
        self.assertTrue(torch.equal(get_cu_seqlens_from_position_ids(pos_2d), expected))
        self.assertTrue(torch.equal(get_cu_seqlens_from_position_ids(pos_4d), expected))

    def test_get_cu_seqlens_fallback_when_no_zero_exists(self):
        pos_2d = torch.tensor([[5, 6, 7]], dtype=torch.long)
        expected = torch.tensor([0, 3], dtype=torch.long)
        self.assertTrue(torch.equal(get_cu_seqlens_from_position_ids(pos_2d), expected))

    def test_packed_detection_consistent_for_2d_and_4d(self):
        sp = SequenceParallel()
        packed_2d = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
        packed_4d = torch.stack([packed_2d, packed_2d + 10, packed_2d + 20, packed_2d + 30], dim=0)
        self.assertTrue(sp._is_packed_position_ids(packed_2d))
        self.assertTrue(sp._is_packed_position_ids(packed_4d))

        non_packed_2d = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long)
        non_packed_4d = torch.stack(
            [non_packed_2d, non_packed_2d + 10, non_packed_2d + 20, non_packed_2d + 30],
            dim=0,
        )
        self.assertFalse(sp._is_packed_position_ids(non_packed_2d))
        self.assertFalse(sp._is_packed_position_ids(non_packed_4d))

    def test_split_attention_mask_works_for_2d_and_4d(self):
        sp = SequenceParallel()
        sp.world_size = 2
        sp.sp_world_size = 2

        def fake_split(tensor, dim: int, position_ids=None):
            return torch.split(tensor, tensor.shape[dim] // 2, dim=dim)[0].contiguous()

        sp.split = fake_split  # type: ignore[assignment]

        real_pos = torch.arange(6).view(1, 6)
        mask_2d = torch.ones((2, 6), dtype=torch.int64)
        split_2d = sp._split_attention_mask(mask_2d, seq_len=6, real_position_ids=real_pos)
        self.assertEqual(split_2d.shape, (2, 3))

        mask_4d = torch.ones((2, 1, 6, 6), dtype=torch.bool)
        split_4d = sp._split_attention_mask(mask_4d, seq_len=6, real_position_ids=real_pos)
        self.assertEqual(split_4d.shape, (2, 1, 3, 3))

    def test_prepare_inputs_stores_text_position_ids(self):
        sp = SequenceParallel()
        position_ids = torch.arange(2 * 6).view(1, 2, 6).expand(4, -1, -1).clone()
        inputs = {
            'input_ids': torch.zeros((2, 6), dtype=torch.long),
            'position_ids': position_ids,
        }
        sp.prepare_inputs(inputs)
        self.assertTrue(torch.equal(sp.real_position_ids, position_ids))
        self.assertTrue(torch.equal(sp.text_position_ids, position_ids[0]))


if __name__ == '__main__':
    unittest.main()
