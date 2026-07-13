from twinkle.model.multi_lora import MultiLora


def test_check_length_allows_batch_when_each_sample_fits():
    multi_lora = MultiLora(max_length=4)

    multi_lora.check_length([
        {'input_ids': [1, 2, 3]},
        {'input_ids': [4, 5, 6]},
    ])


def test_check_length_rejects_single_oversized_sample():
    multi_lora = MultiLora(max_length=4)

    try:
        multi_lora.check_length([{'input_ids': [1, 2, 3, 4, 5]}])
    except ValueError as exc:
        assert 'Input length 5 exceeds max_length 4' in str(exc)
    else:
        raise AssertionError('expected oversized sample to fail')
