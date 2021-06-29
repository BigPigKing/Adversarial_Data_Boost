import abc
import torch

from tqdm import tqdm
from typing import Dict
from collections import Counter
from overrides import overrides
from allennlp.nn.util import move_to_device
from torch.utils.tensorboard import SummaryWriter


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self
    ):
        return NotImplemented


class ReinforceTrainer(Trainer):
    def __init__(
        self,
        train_model: torch.nn.Module,
        is_writer: bool = False,
        is_save: bool = False
    ):
        super(ReinforceTrainer, self).__init__()
        self.train_model = train_model

        if is_writer:
            self.writer = SummaryWriter()
        else:
            self.writer = None

        self.is_save = is_save
        self.record_step = 0

        self.GPU = next(train_model.parameters()).get_device()

    def _record(
        self,
        step: int,
        batch_output_dict: Dict,
        batch_size: int
    ):
        self.writer.add_scalar(
            "Loss",
            batch_output_dict["loss"] / batch_size,
            step
        )
        self.writer.add_scalar(
            "Reward",
            batch_output_dict["reward"] / batch_size,
            step
        )
        self.writer.add_text(
            "Origin",
            batch_output_dict["origin_sentences"][-1],
            step
        )
        self.writer.add_text(
            "Augment",
            batch_output_dict["augment_sentences"][-1],
            step
        )
        action_str = [str(x) for x in batch_output_dict["actions"][-1]]
        action_str = ' '.join(action_str)
        self.writer.add_text(
            "Action",
            action_str,
            step
        )

    def _fit_epoch(
        self,
        batch_size: int,
        data_loader: torch.utils.data.DataLoader,
    ):
        batch_output_dict = {
            "loss": 0.0,
            "reward": 0.0,
            "actions": [],
            "total_actions": [],
            "origin_sentences": [],
            "augment_sentences": []
        }

        for episode_idx, episode in enumerate(tqdm(data_loader)):
            # feedforward and get loss
            if self.GPU >= 0:
                episode = move_to_device(episode, self.GPU)
            else:
                pass

            output_dict = self.train_model.forward(episode)

            # update batch dict
            batch_output_dict["loss"] += output_dict["loss"]
            batch_output_dict["reward"] += output_dict["ep_reward"]
            batch_output_dict["origin_sentences"].append(output_dict["origin_sentence"])
            batch_output_dict["augment_sentences"].append(output_dict["augment_sentence"])
            batch_output_dict["actions"].append(output_dict["actions"])
            batch_output_dict["total_actions"] += output_dict["actions"]

            # batch updating
            if (episode_idx+1) % batch_size == 0:
                # Record
                print(self.record_step)
                print(output_dict["origin_sentence"])
                print(output_dict["augment_sentence"])
                print(output_dict["actions"])
                print(output_dict["ep_reward"])
                print(dict(Counter(output_dict["total_actions"])))

                if self.writer is None:
                    pass
                else:
                    self._record(self.record_step, batch_output_dict, batch_size)

                self.record_step += 1

                # Optimize
                self.train_model.optimize(batch_output_dict["loss"] / batch_size)
                print(batch_output_dict["reward"] / batch_size)

                # Initialize
                batch_output_dict = {
                    "loss": 0.0,
                    "reward": 0.0,
                    "actions": [],
                    "total_actions": [],
                    "origin_sentences": [],
                    "augment_sentences": []
                }

                # Save
                if self.is_save is True:
                    torch.save(
                        self.train_model.policy.state_dict(),
                        "model_record/reinforce_model_weights/policy" + str(self.record_step) + ".pkl"
                    )

    def fit(
        self,
        epochs: int,
        batch_size: int,
        data_loader: torch.utils.data.DataLoader,
    ):
        for epoch in tqdm(range(epochs)):
            self.train_model.train()
            self._fit_epoch(batch_size, data_loader)


class TextTrainer(Trainer):
    def __init__(
        self,
        text_trainer_params: Dict,
        train_model: torch.nn.Module,
    ):
        super(TextTrainer, self).__init__()
        self.train_model = train_model
        self.is_save = text_trainer_params["is_save"]
        self.accumulated_step = text_trainer_params["accumulated_step"]

        self.augment_loss_multiplier = 0.6
        self.consistency_loss_multiplier = 0.99
        self.contrastive_loss_multiplier = 0.3

        self.GPU = next(train_model.parameters()).get_device()

    def _fit_valid(
        self,
        valid_data_loader: torch.utils.data.DataLoader
    ):
        num_of_batch = 0
        total_loss = 0.0
        total_labels = []
        total_predicts = []

        for batch_idx, batch in enumerate(valid_data_loader):
            if self.GPU >= 0:
                batch = move_to_device(batch, self.GPU)
            else:
                pass

            output_dict = self.train_model.forward(batch, is_finetune=False)

            num_of_batch += 1
            total_labels.append(batch["label"])
            total_predicts.append(output_dict["predicts"])
            total_loss += output_dict["classification_loss"].item()

        total_labels = torch.cat(total_labels, 0)
        total_predicts = torch.cat(total_predicts, 0)

        avg_loss = total_loss / num_of_batch
        avg_acc = torch.true_divide(torch.sum(total_labels == total_predicts), total_labels.shape[0])

        return avg_loss, avg_acc

    def _fit_train(
        self,
        train_data_loader: torch.utils.data.DataLoader,
        is_finetune: bool
    ):
        num_of_batch = 0
        total_loss = 0.0
        total_origin_loss = 0.0
        total_augment_loss = 0.0
        total_consistency_loss = 0.0
        total_labels = []
        total_predicts = []

        for batch_idx, batch in enumerate(tqdm(train_data_loader)):
            if self.GPU >= 0:
                batch = move_to_device(batch, self.GPU)
            else:
                pass

            output_dict = self.train_model.forward(
                batch,
                is_finetune
            )

            # Optimize
            if is_finetune is False:
                batch_loss = output_dict["classification_loss"]
                total_origin_loss += output_dict["classification_loss"]
            else:
                batch_loss = output_dict["origin_classification_loss"] + self.augment_loss_multiplier * output_dict["total_augment_loss"] + self.consistency_loss_multiplier * output_dict["total_consistency_loss"]  # noqa
                total_origin_loss += output_dict["origin_classification_loss"]
                total_augment_loss += output_dict["total_augment_loss"]
                total_consistency_loss += output_dict["total_consistency_loss"]

            # Accumulated
            if (batch_idx+1) % self.accumulated_step == 0:
                self.train_model.optimize(
                    batch_loss / self.accumulated_step,
                    [self.train_model.optimizer],
                    is_step=True
                )
            else:
                self.train_model.optimize(
                    batch_loss / self.accumulated_step,
                    [self.train_model.optimizer],
                    is_step=False
                )

            num_of_batch += 1
            total_labels.append(batch["label"])
            total_predicts.append(output_dict["predicts"])
            total_loss += batch_loss.item()

        self.train_model.optimizer.zero_grad()

        total_labels = torch.cat(total_labels, 0)
        total_predicts = torch.cat(total_predicts, 0)

        avg_loss = total_loss / num_of_batch
        avg_origin_loss = total_origin_loss / num_of_batch
        avg_augment_loss = total_augment_loss / num_of_batch
        avg_consistency_loss = total_consistency_loss / num_of_batch
        avg_acc = torch.true_divide(torch.sum(total_labels == total_predicts), total_labels.shape[0])

        loss_dict = {
            "avg_loss": avg_loss,
            "avg_origin_loss": avg_origin_loss,
            "avg_augment_loss": avg_augment_loss,
            "avg_consistency_loss": avg_consistency_loss
        }

        return loss_dict, avg_acc

    @overrides
    def fit(
        self,
        epochs: int,
        is_finetune: bool,
        train_data_loader: torch.utils.data.DataLoader,
        valid_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader = None
    ):
        for epoch in tqdm(range(epochs)):
            # Do training
            self.train_model.train()
            loss_dict, train_avg_acc = self._fit_train(train_data_loader, is_finetune)

            # Do validation
            self.train_model.eval()
            with torch.no_grad():
                valid_avg_loss, valid_avg_acc = self._fit_valid(valid_data_loader)

            # Do testing
            self.train_model.eval()
            with torch.no_grad():
                test_avg_loss, test_avg_acc = self._fit_valid(test_data_loader)

            print("Epochs                   : {}".format(epoch))
            print("Training Total Loss      : {:.5f}".format(loss_dict["avg_loss"]))
            print("Training Origin Loss     : {:.5f}".format(loss_dict["avg_origin_loss"]))
            print("Training Augment Loss    : {:.5f}".format(loss_dict["avg_augment_loss"]))
            print("Training Consistency Loss: {:.5f}".format(loss_dict["avg_consistency_loss"]))
            print("Training Acc             : {:.5f}".format(train_avg_acc))
            print("Validation Loss          : {:.5f}".format(valid_avg_loss))
            print("Validation Acc           : {:.5f}".format(valid_avg_acc))
            print("Testing Loss             : {:.5f}".format(test_avg_loss))
            print("Testing Acc              : {:.5f}".format(test_avg_acc))
            print("----------------------------------------------")

        if self.is_save is True:
            torch.save(self.train_model.embedder.state_dict(), "model_record/text_model_weights/embedder.pkl")
            torch.save(self.train_model.encoder.state_dict(), "model_record/text_model_weights/encoder.pkl")
            torch.save(self.train_model.classifier.state_dict(), "model_record/text_model_weights/classifier.pkl")
        else:
            pass


class OverallTrainer(Trainer):
    def __init__(
        self,
        text_trainer: Trainer,
        reinforce_trainer: Trainer
    ):
        self.text_trainer = text_trainer
        self.reinforce_trainer = reinforce_trainer


def main():
    pass


if __name__ == '__main__':
    main()
