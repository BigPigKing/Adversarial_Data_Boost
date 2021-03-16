import abc
import math
import torch
import random

from .utils import pad_text_tensor_list, unpad_text_field_tensors
from typing import Dict
from overrides import overrides
from nltk.corpus import wordnet
from allennlp.data import Vocabulary


class Augmenter(metaclass=abc.ABCMeta):
    def __init__(
        self,
        padded_idx: int = 0
    ):
        self.padded_idx = padded_idx

    @abc.abstractmethod
    def _augment(
        self,
        text_tensor: torch.Tensor
    ):
        return NotImplemented

    def augment(
        self,
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
    ):
        augment_text_tensor_list = []
        text_tensor_list = unpad_text_field_tensors(text_field_tensors)

        for text_tensor in text_tensor_list:
            augment_text_tensor = self._augment(text_tensor)
            augment_text_tensor_list.append(augment_text_tensor)

        return pad_text_tensor_list(augment_text_tensor_list)


class DeleteAugmenter(Augmenter):
    def __init__(
        self,
        padded_idx: int = 0,
        magnitude: float = 0.1
    ):
        super(DeleteAugmenter, self).__init__(padded_idx=padded_idx)
        self.magnitude = magnitude

    @overrides
    def _augment(
        self,
        text_tensor: torch.Tensor
    ) -> torch.Tensor:
        if len(text_tensor) == 1:
            return text_tensor.detach().clone()
        else:
            # Get delete word information
            num_of_del_word = math.floor(len(text_tensor) * self.magnitude)
            del_word_idxs = random.sample(range(len(text_tensor)), num_of_del_word)
            del_word_idxs.sort()
            text_list = text_tensor.tolist()

            # Loop to delete word
            for del_word_idx in reversed(del_word_idxs):
                del text_list[del_word_idx]

            return torch.tensor(text_list).to(text_tensor.get_device())


class SwapAugmenter(Augmenter):
    def __init__(
        self,
        padded_idx: int = 0,
        magnitude: int = 1
    ):
        super(SwapAugmenter, self).__init__(padded_idx=padded_idx)
        self.magnitude = magnitude

    @overrides
    def _augment(
        self,
        text_tensor: torch.Tensor
    ) -> torch.Tensor:
        # Prepare tensor for augmentation
        augment_text_tensor = text_tensor.detach().clone()

        if len(text_tensor) == 1:
            return augment_text_tensor
        else:
            # Sample swap index
            first_idx, second_idx = random.sample(range(len(augment_text_tensor)), 2)

            temp = augment_text_tensor[first_idx].clone()
            augment_text_tensor[first_idx] = augment_text_tensor[second_idx]
            augment_text_tensor[second_idx] = temp

        return augment_text_tensor


class ReplaceAugmenter(Augmenter):
    def __init__(
        self,
        vocab: Vocabulary,
        padded_idx: int = 0,
        magnitude: int = 1,
    ):
        super(ReplaceAugmenter, self).__init__(padded_idx=padded_idx)
        self.magnitude = magnitude
        self.vocab = vocab

    def _get_synonyms(
        self,
        token: str
    ):
        synonyms = set()

        for syn in wordnet.synsets(token):
            for synonym_lemma in syn.lemmas():
                synonym = synonym_lemma.name().replace('_', ' ').replace('-', ' ').lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(tuple(synonym.split()))

        synonyms = list(synonyms)

        if tuple([token]) in synonyms:
            synonyms.remove(tuple([token]))
        else:
            pass

        return synonyms

    def _get_replace_synonym(
        self,
        token_id: int
    ):
        token = self.vocab.get_token_from_index(token_id)
        synonyms = self._get_synonyms(token)

        if synonyms:
            return random.choice(synonyms)
        else:
            return []

    @overrides
    def _augment(
        self,
        text_tensor: torch.Tensor
    ):
        augment_text_list = text_tensor.tolist()
        replace_idx = random.sample(range(len(augment_text_list)), 1)[0]

        replace_synonym = self._get_replace_synonym(augment_text_list[replace_idx])

        if replace_synonym:
            for idx, synonym_token in enumerate(replace_synonym):
                synonym_idx = self.vocab.get_token_index(synonym_token)

                if idx == 0:
                    augment_text_list[replace_idx] = synonym_idx
                else:
                    augment_text_list.insert(replace_idx + idx, synonym_idx)
        else:
            pass

        return torch.tensor(augment_text_list).to(text_tensor.get_device())


class InsertAugmenter(ReplaceAugmenter):
    def __init__(
        self,
        vocab: Vocabulary,
        padded_idx: int = 0,
        magnitude: int = 1
    ):
        super(InsertAugmenter, self).__init__(padded_idx=padded_idx, vocab=vocab, magnitude=magnitude)

    @overrides
    def _augment(
        self,
        text_tensor: torch.tensor
    ):
        augment_text_list = text_tensor.tolist()
        replace_idx = random.sample(range(len(augment_text_list)), 1)[0]

        replace_synonym = self._get_replace_synonym(augment_text_list[replace_idx])

        if replace_synonym:
            for idx, synonym_token in enumerate(replace_synonym):
                synonym_idx = self.vocab.get_token_index(synonym_token)
                augment_text_list.insert(replace_idx + idx, synonym_idx)
        else:
            pass

        return torch.tensor(augment_text_list).to(text_tensor.get_device())


class IdentityAugmenter(Augmenter):
    def __init__(
        self,
        padded_idx: int = 0
    ):
        super(IdentityAugmenter, self).__init__(padded_idx=padded_idx)

    @overrides
    def _augment(
        self,
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
    ):
        return text_field_tensors


def main():
    pass


if __name__ == '__main__':
    main()
