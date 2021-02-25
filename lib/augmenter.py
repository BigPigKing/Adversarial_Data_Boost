import abc
import math
import torch
import random

from .utils import pad_text_tensor_list, unpad_text_field_tensors
from typing import Dict
from overrides import overrides


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
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
        proportion: float = 0.1
    ):
        augment_text_tensor_list = []
        text_tensor_list = unpad_text_field_tensors(text_field_tensors)

        for text_tensor in text_tensor_list:
            if len(text_tensor) == 1:
                augment_text_tensor_list.append(text_tensor.detach().clone())
            else:
                # Get delete word information
                num_of_del_word = math.floor(len(text_tensor) * proportion)
                del_word_idxs = random.sample(range(0, len(text_tensor - 1)), num_of_del_word)

                # Loop to delete word
                augment_text_tensor = text_tensor.detach().clone()
                for del_word_idx in del_word_idxs:
                    augment_text_tensor[del_word_idx] = self.padded_idx

                augment_text_tensor_list.append(augment_text_tensor)

        return pad_text_tensor_list(augment_text_tensor_list)


class SwapAugmenter(Augmenter):
    def __init__(
        self
    ):
        return

    def _swap(
        self,
        text_tensor: torch.Tensor
    ) -> torch.Tensor:
        augment_text_tensor = text_tensor.detach().clone()
        first_idx, second_idx = random.sample(range(len(text_tensor)), 2)

        temp = augment_text_tensor[first_idx].clone()
        augment_text_tensor[first_idx] = augment_text_tensor[second_idx]
        augment_text_tensor[second_idx] = temp

        return text_tensor

    @overrides
    def augment(
        self,
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]],
        num_of_augmentation: int = 1
    ):
        augment_text_tensor_list = []
        text_tensor_list = unpad_text_field_tensors(text_field_tensors)

        for text_tensor in text_tensor_list:
            if len(text_tensor) == 1:
                augment_text_tensor_list.append(text_tensor.detach().clone())
            else:
                augment_text_tensor_list.append(self._swap(text_tensor))

        return pad_text_tensor_list(augment_text_tensor_list)


class ReplaceAugmenter(Augmenter):
    def __init__():
        pass


def main():
    pass


if __name__ == '__main__':
    main()
