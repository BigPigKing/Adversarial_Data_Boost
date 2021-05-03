import torch

from typing import Dict
from overrides import overrides
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask


class SentimentModel(torch.nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: torch.nn.Module,
        encoder: torch.nn.ModuleList,
        classifier: torch.nn.ModuleList,
        sentiment_model_params: Dict
    ):
        super(SentimentModel, self).__init__()
        # Model Augmenter
        self.vocab = vocab

        # Small Module
        self.embedder = embedder
        self.encoder = encoder
        self.classifier = classifier

        # Loss initiailization
        self.classification_criterion = sentiment_model_params["criterions"]["classification_criterion"]
        self.contrastive_criterion = sentiment_model_params["criterions"]["contrastive_criterion"]

        # Optimizer initialization
        self.optimizer = sentiment_model_params["optimizer"]["select_optimizer"](
            self.parameters(),
            lr=sentiment_model_params["optimizer"]["lr"]
        )

        # Scheduler inititailzation
        if sentiment_model_params["scheduler"]["select_scheduler"] != "none":
            # self.scheduler = sentiment_model_params["scheduler"]["select_scheduler"](self.optimizer)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        else:
            self.scheduler = None

        # Evaluate initialization
        self.accuracy = sentiment_model_params["evaluation"]

    @overrides
    def forward(
        self,
        token_X,
        label_Y,
    ) -> Dict:
        output_dict = {}

        # Embedded first
        embed_X = self.embedder(token_X)

        # Get token mask for speed up
        tokens_mask = get_text_field_mask(token_X)

        # Encode
        encode_X = self.encoder(embed_X, tokens_mask)

        # Classfiy
        pred_Y = self.classifier(encode_X)

        classification_loss = self.classification_criterion(pred_Y, label_Y)
        output_dict["classification_loss"] = classification_loss
        output_dict["predicts"] = torch.argmax(pred_Y, dim=1)

        return output_dict

    def optimize(
        self,
        loss,
        optimizers
    ):
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()


def main():
    pass


if __name__ == '__main__':
    main()
