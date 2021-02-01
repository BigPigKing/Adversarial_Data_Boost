import abc
import torch

from typing import Dict
from overrides import overrides
from allennlp.nn.util import get_token_ids_from_text_field_tensors


class Augmenter(metaclass=abc.ABCMeta):
    def __init__(
        self,
    ):
        pass

    @abc.abstractmethod
    def augment(
        self
    ):
        return NotImplemented


class DeleteAugmenter(Augmenter):
    def __init__(
        self,
        padded_idx: int,
    ):
        self.padded_idx = padded_idx

    @overrides
    def augment(
        self,
        ids_dict: Dict[str, Dict[str, torch.Tensor]],
        prob: float
    ):
        token_ids = get_token_ids_from_text_field_tensors(ids_dict)

        # Every token_id means a sentence based token id list
        # token_ids mean all the sentences in the batch
        for token_id in token_ids:
            non_padded_token_id = token_id[token_id != self.padded_idx]


def main():
    pass


if __name__ == '__main__':
    main()
