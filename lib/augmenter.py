import abc
import copy
import math
import torch
import random

from .utils import pad_text_tensor_list, unpad_text_field_tensors
from .tokenizer import WordTokenizer
from typing import Dict, List
from overrides import overrides
from nltk.corpus import wordnet
from allennlp.data import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.modules.token_embedders import Embedding


class Augmenter(object):
    def __init__(
        self,
        vocab: Vocabulary,
        dataset_dict: Dict,
        tokenizer=None,
        padded_idx: int = 0
    ):
        self.detokenizer = dataset_dict["dataset_reader"]._tokenizer
        self.vocab = dataset_dict["dataset_reader"]._vocab
        self.indexer = dataset_dict["dataset_reader"]._indexers["tokens"]
        self.is_transformer = dataset_dict["dataset_reader"].is_transformer
        self.padded_idx = padded_idx

        try:  # if non-transformer
            self.detokenizer.index_with(vocab)
        except AttributeError:
            pass
        self.tokenizer = tokenizer or WordTokenizer()

    def _get_decode_str(
        self,
        token_ids: torch.Tensor
    ):
        try:  # This is for the non-transformer tokenizer, because transformer tokenizer in allennlp doesn't have decode
            decode_str = self.detokenizer.decode(token_ids.tolist())
        except AttributeError:
            decode_str = self.detokenizer.tokenizer.decode(
                token_ids.tolist(),
                skip_special_tokens=True
            )

        return decode_str

    def _get_encode_token_ids(
        self,
        input_str: str
    ):
        try:  # Allennlp pretrained transformer tokenizer doesn't have encode attribute but others have
            token_ids = self.detokenizer.encode(
                input_str
            )
        except AttributeError:  # Allennlp indexer output is different [Warn]
            tokens = self.detokenizer.tokenize(
                input_str
            )
            token_ids = self.indexer.tokens_to_indices(
                tokens,
                self.vocab
            )

        return token_ids

    @abc.abstractmethod
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        return NotImplemented

    def _augment(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        # Decode to original string
        decode_str = self._get_decode_str(
            token_ids
        )

        # Tokenize the original string
        decode_tokens = self.tokenizer.tokenize(
            decode_str
        )

        # Action
        augmented_tokens = self._action(
            copy.deepcopy(decode_tokens)
        )

        # Get Augmented String
        augmented_str = self.tokenizer.detokenize(
            augmented_tokens
        )

        # Encode to token_ids
        augmented_token_ids = self._get_encode_token_ids(
            augmented_str
        )

        # print("OriginTokenIds")
        # print(token_ids)
        # print()
        # print("DecodeStr")
        # print(decode_str)
        # print()
        # print("DecodeTokens")
        # print(decode_tokens)
        # print()
        # print("AugmentTokens")
        # print(augmented_tokens)
        # print()
        # print("AugmentStr")
        # print(augmented_str)
        # print()
        # print("AugmentTokenIds")
        # print(augmented_token_ids)
        # print()
        # print("LEN TOKEN ID")
        # print(len(token_ids))
        # print("LEN AUGMENTED")
        # print(len(augmented_token_ids["token_ids"]))

        return augmented_token_ids

    def augment(
        self,
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]]
    ):
        augment_text_tensor_list = []
        text_tensor_list = unpad_text_field_tensors(text_field_tensors, self.is_transformer)

        for text_tensor in text_tensor_list:
            augment_text_tensor = self._augment(text_tensor)
            augment_text_tensor_list.append(augment_text_tensor)

        return move_to_device(
            pad_text_tensor_list(
                augment_text_tensor_list,
                self.is_transformer,
                indexer=self.indexer
            ),
            text_tensor_list[0].get_device()
        )


class DeleteAugmenter(Augmenter):
    def __init__(
        self,
        delete_augmenter_params: Dict,
        vocab: Vocabulary,
        dataset_dict: Dict
    ):
        super(DeleteAugmenter, self).__init__(
            vocab,
            dataset_dict
        )
        self.magnitude = delete_augmenter_params["magnitude"]

    @overrides
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        # print("!!!!DELETE")
        if len(tokens) == 1:
            return tokens
        else:
            pass

        # Get Delete Word Indexes
        num_of_del_word = max(1, math.floor(len(tokens) * self.magnitude))
        del_word_idxs = random.sample(range(len(tokens)), num_of_del_word)

        if len(del_word_idxs) > len(tokens) + 1:
            return tokens
        else:
            del_word_idxs.sort()

            # Delete
            for del_word_idx in reversed(del_word_idxs):
                del tokens[del_word_idx]

        return tokens


class SwapAugmenter(Augmenter):
    def __init__(
        self,
        swap_augmenter_params: Dict,
        vocab: Vocabulary,
        dataset_dict: Dict
    ):
        super(SwapAugmenter, self).__init__(
            vocab,
            dataset_dict
        )
        self.magnitude = swap_augmenter_params["magnitude"]

    @overrides
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        if len(tokens) == 1:
            return tokens
        else:
            # Sample swap index
            select_idxs = random.sample(range(len(tokens)), max(len(tokens) * self.magnitude, 2))
            swap_idxs = copy.deepcopy(select_idxs)
            random.shuffle(select_idxs)
            swap_tokens = [copy.deepcopy(tokens[x]) for x in swap_idxs]

            for idx, (select_idx, swap_idx) in enumerate(zip(select_idxs, swap_idxs)):
                tokens[select_idx] = swap_tokens[idx]

            return tokens


class ReplaceAugmenter(Augmenter):
    def __init__(
        self,
        replace_augmenter_params: Dict,
        vocab: Vocabulary,
        dataset_dict: Dict
    ):
        super(ReplaceAugmenter, self).__init__(
            vocab,
            dataset_dict
        )
        self.magnitude = replace_augmenter_params["magnitude"]

    def _find_synonyms(
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

    def _get_synonyms(
        self,
        token: str
    ):
        synonyms = self._find_synonyms(token)

        return synonyms

    def _get_replace_synonym(
        self,
        token: str
    ) -> tuple:
        synonyms = self._get_synonyms(token)

        if synonyms:
            return random.choice(synonyms)
        else:
            return None

    @overrides
    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        availiable_token_idxs = []
        availiable_synonyms = []

        for idx in range(len(tokens)):
            replace_synonym = self._get_replace_synonym(
                tokens[idx]
            )

            if replace_synonym:
                availiable_token_idxs.append(idx)
                availiable_synonyms.append(replace_synonym)
            else:
                pass

        if len(availiable_synonyms) == 0:
            return tokens

        replace_idxs = random.sample(
            range(len(availiable_token_idxs)),
            max(int(len(availiable_token_idxs) * self.magnitude), 1)
        )

        for replace_idx in replace_idxs:
            replace_synonym = availiable_synonyms[replace_idx]

            for synonym_token_idx, synonym_token in enumerate(replace_synonym):
                if synonym_token_idx == 0:
                    tokens[availiable_token_idxs[replace_idx]] = synonym_token
                else:
                    tokens.insert(
                        availiable_token_idxs[replace_idx] + synonym_token_idx,
                        synonym_token
                    )

        return tokens


class InsertAugmenter(ReplaceAugmenter):
    def __init__(
        self,
        insert_augmenter_params: Dict,
        vocab: Vocabulary,
        dataset_dict: Dict
    ):
        super(InsertAugmenter, self).__init__(
            insert_augmenter_params,
            vocab,
            dataset_dict
        )

    def _action(
        self,
        tokens: List[str]
    ) -> List[str]:
        availiable_token_idxs = []
        availiable_synonyms = []

        for idx in range(len(tokens)):
            replace_synonym = self._get_replace_synonym(
                tokens[idx]
            )

            if replace_synonym:
                availiable_token_idxs.append(idx)
                availiable_synonyms.append(replace_synonym)
            else:
                pass

        if len(availiable_synonyms) == 0:
            return tokens

        replace_idxs = random.sample(
            range(len(availiable_token_idxs)),
            max(int(len(availiable_token_idxs) * self.magnitude), 1)
        )

        for replace_idx in replace_idxs:
            replace_synonym = availiable_synonyms[replace_idx]

            for synonym_token_idx, synonym_token in enumerate(replace_synonym):
                tokens.insert(
                    availiable_token_idxs[replace_idx] + synonym_token_idx,
                    synonym_token
                )

        return tokens


class OldInsertAugmenter(ReplaceAugmenter):
    def __init__(
        self,
        vocab: Vocabulary,
        embedding_layer: Embedding,
        synonym_dict: Dict,
        insert_augmenter_params: Dict
    ):
        super(OldInsertAugmenter, self).__init__(
            vocab,
            embedding_layer,
            synonym_dict,
            insert_augmenter_params
        )

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

                if synonym_idx == self.oov_idx:
                    synonym_idx = self.vocab.add_token_to_namespace(synonym_token)
                    self.embedding_layer.token_embedders["tokens"].extend_vocab(self.vocab)
                else:
                    pass

                augment_text_list.insert(replace_idx + idx, synonym_idx)
        else:
            pass

        return torch.tensor(augment_text_list).to(text_tensor.get_device())


class IdentityAugmenter(Augmenter):
    def __init__(
        self,
        is_transformer: bool,
        padded_idx: int = 0
    ):
        return
        # super(IdentityAugmenter, self).__init__(is_transformer, padded_idx=padded_idx)

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
