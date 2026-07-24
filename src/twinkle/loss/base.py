# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import InputFeature, LossOutput, ModelOutput


class Loss:

    require_logits = False
    require_entropy = False
    require_logps = True

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs) -> LossOutput:
        ...

    def micro_batch_scale(self, inputs: list[InputFeature], indices: list[int]) -> float:
        if len(indices) == len(inputs):
            return 1.0
        raise NotImplementedError(
            f'{self.__class__.__name__} does not define how independently computed '
            'micro-batch losses should be combined')

    @staticmethod
    def token_mean_micro_batch_scale(
        inputs: list[InputFeature],
        indices: list[int],
        *,
        ignore_index: int,
    ) -> float:

        def count_tokens(model_input: InputFeature) -> int:
            labels = model_input['labels']
            if hasattr(labels, 'ne'):
                return int(labels.ne(ignore_index).sum().item())
            return sum(int(token != ignore_index) for token in labels)

        token_counts = [count_tokens(model_input) for model_input in inputs]
        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return 0.0
        return sum(token_counts[index] for index in indices) / total_tokens
