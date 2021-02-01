import torch

from overrides import overrides
from typing import List, Dict
from .augmenter import Augmenter
from allennlp.nn.util import get_token_ids_from_text_field_tensors, get_text_field_mask


class Policy(torch.nn.Module):
    def __init__(
        self,
    ):
        super(Policy, self).__init__()

    @overrides
    def forward(
        self,
        encoded_sent_state: torch.Tensor
    ):



class REINFORCER(torch.nn.Module):
    def __init__(
        self,
        embedder: torch.nn.Module,
        encoder: torch.nn.Module,
        augmenter_list: List[Augmenter]
    ):
        super(REINFORCER, self).__init__()
        self.embedder = embedder
        self.encoder = encoder
        self.augmenter_list = augmenter_list

    def _get_token_of_sents(
        self,
        token_ids: Dict[str, Dict[str, torch.Tensor]]
    ):
        token_of_sents = get_token_ids_from_text_field_tensors(token_ids)

        return token_of_sents

    def _wrap_token_of_sent(
        self,
        token_of_sent: torch.Tensor
    ):
        wrapped_token_of_sent = torch.stack([token_of_sent])

        print(wrapped_token_of_sent)
        print(wrapped_token_of_sent.shape)

        return {"tokens": {"tokens": wrapped_token_of_sent}}

    @overrides
    def forward(
        self,
        wrapped_token_of_sent: Dict[str, Dict[str, torch.Tensor]]
    ):
        # Embedded first
        embedded_sent = self.embedder(wrapped_token_of_sent)

        # get token mask
        sent_mask = get_text_field_mask(wrapped_token_of_sent)

        # Encode
        encoded_sent = self.encoder(embedded_sent, sent_mask)

        return encoded_sent

    def augment_batch(
        self,
        token_ids: Dict[str, Dict[str, torch.Tensor]]
    ):
        token_of_sents = self._get_token_of_sents(token_ids)

        for token_of_sent in token_of_sents:
            wrapped_token_of_sent = self._wrap_token_of_sent(token_of_sent)
            encoded_sent = self.forward(wrapped_token_of_sent)

            print(encoded_sent)
            print(encoded_sent.shape)


def main():
    pass


if __name__ == '__main__':
    main()
