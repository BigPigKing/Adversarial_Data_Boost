import torch

from .loss import JsdCrossEntropy
from typing import List, Dict
from overrides import overrides
from transformers import get_linear_schedule_with_warmup
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask


class SentimentModel(torch.nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: torch.nn.Module,
        encoder: torch.nn.ModuleList,
        classifier: torch.nn.ModuleList,
        sentiment_model_params: Dict,
        dataset_dict: Dict,
        is_finetune: bool = False
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
        self.consistency_criterion = JsdCrossEntropy()

        # Optimizer initialization
        self.optimizer = sentiment_model_params["optimizer"]["select_optimizer"](
            self.parameters(),
            lr=sentiment_model_params["optimizer"]["lr"]
        )

        # Scheduler inititailzation
        if sentiment_model_params["scheduler"]["select_scheduler"] != "none":
            # self.scheduler = sentiment_model_params["scheduler"]["select_scheduler"](self.optimizer)
            # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=sentiment_model_params["scheduler"]["warmup_steps"],
                num_training_steps=sentiment_model_params["scheduler"]["training_steps"]
            )
        else:
            self.scheduler = None

        # Evaluate initialization
        self.accuracy = sentiment_model_params["evaluation"]

        # Clip initialization
        self.is_clip = sentiment_model_params["clip_grad"]["is_clip"]
        self.max_norm = sentiment_model_params["clip_grad"]["max_norm"]
        self.norm_type = sentiment_model_params["clip_grad"]["norm_type"]

        self.is_finetune = is_finetune

        # Field Names
        self.text_field_names = dataset_dict["dataset_reader"].field_names["text"]
        self.label_field_names = dataset_dict["dataset_reader"].field_names["label"]
        self.augment_field_names = None

    def set_augment_field_names(
        self,
        augment_field_names: List[str]
    ):
        self.augment_field_names = augment_field_names

    def _get_classification_loss_and_predicts(
        self,
        token_X: torch.Tensor,
        label_Y: torch.Tensor
    ) -> Dict:
        # Embedded first
        embed_X = self.embedder(token_X)

        # Get token mask for speed up
        tokens_mask = get_text_field_mask(token_X)

        # Encode
        encode_X = self.encoder(embed_X, tokens_mask)

        # Classfiy
        if self.is_finetune:
            pred_Y = self.classifier(encode_X.detach())
        else:
            pred_Y = self.classifier(encode_X)

        # Get all the loss
        classification_loss = self.classification_criterion(
            pred_Y,
            label_Y
        )

        return classification_loss, pred_Y

    def _standard_forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict:
        output_dict = {}

        # Get input from dict
        token_X = batch[self.text_field_names[0]]
        label_Y = batch[self.label_field_names[0]]

        # Get Cross entropy Loss
        classification_loss, predicts = self._get_classification_loss_and_predicts(
            token_X,
            label_Y
        )

        output_dict["classification_loss"] = classification_loss
        output_dict["predicts"] = torch.argmax(predicts, dim=1)

        return output_dict

    def _finetune_forward(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict:
        output_dict = {}

        # Get input from dict
        origin_token_X = batch[self.text_field_names[0]]
        origin_label_Y = batch[self.label_field_names[0]]

        # Get Origin Cross entropy Loss
        origin_classification_loss, origin_predicts = self._get_classification_loss_and_predicts(
            origin_token_X,
            origin_label_Y
        )

        # Get Consistency Loss and Augmented Cross entropy Loss
        assert self.augment_field_names is not None, "Augmented field names is not given!"

        total_augment_loss = 0
        total_consistency_loss = 0
        for augment_field_name in self.augment_field_names:
            augment_classification_loss, augment_predicts = self._get_classification_loss_and_predicts(
                batch[augment_field_name],
                origin_label_Y
            )
            total_augment_loss += augment_classification_loss
            total_consistency_loss += self.consistency_criterion(origin_predicts, augment_predicts)

        output_dict["origin_classification_loss"] = origin_classification_loss
        output_dict["total_augment_loss"] = total_augment_loss / len(self.augment_field_names)
        output_dict["total_consistency_loss"] = total_consistency_loss
        output_dict["predicts"] = torch.argmax(origin_predicts, dim=1)

        return output_dict

    @overrides
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        is_finetune: bool
    ) -> Dict:
        if is_finetune is False:
            output_dict = self._standard_forward(batch)
        else:
            output_dict = self._finetune_forward(batch)

        return output_dict

    def optimize(
        self,
        loss,
        optimizers,
        is_step: bool = True,
    ):
        loss.backward()

        for optimizer in optimizers:
            if self.is_clip is True:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(),
                    self.max_norm,
                    self.norm_type
                )
            else:
                pass

            if is_step is True:
                optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                else:
                    pass
                optimizer.zero_grad()
            else:
                pass


def main():
    pass


if __name__ == '__main__':
    main()
