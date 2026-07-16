import pytest

from twinkle.model.multi_lora import MultiLora


def test_check_length_validates_each_sample_independently():
    multi_lora = MultiLora(max_length=8)

    multi_lora.check_length([
        {'input_ids': list(range(6))},
        {'input_ids': list(range(6))},
    ])


def test_check_length_reports_the_oversized_sample():
    multi_lora = MultiLora(max_length=8)

    with pytest.raises(ValueError, match='Input length 9 exceeds max_length 8 at sample 1'):
        multi_lora.check_length([
            {'input_ids': list(range(4))},
            {'input_ids': list(range(9))},
        ])


def test_check_length_accepts_one_input_feature():
    multi_lora = MultiLora(max_length=8)

    multi_lora.check_length({'input_ids': list(range(8))})
