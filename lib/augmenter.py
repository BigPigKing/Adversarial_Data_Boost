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
    ):
        return ids_dict


def main():
    pass


if __name__ == '__main__':
    main()
