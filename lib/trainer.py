import abc
import torch

from tqdm import tqdm
from typing import Dict
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
        batch_output_dict: Dict
    ):
        self.writer.add_scalar("Loss", batch_output_dict["loss"], step)
        self.writer.add_scalar("Reward", batch_output_dict["reward"], step)
        self.writer.add_text("Origin", batch_output_dict["origin_sentences"][0], step)
        self.writer.add_text("Augment", batch_output_dict["augment_sentences"][0], step)
        action_str = [str(x) for x in batch_output_dict["actions"]]
        action_str = ' '.join(action_str)
        self.writer.add_text("Action", action_str, step)

    def _fit_epoch(
        self,
        batch_size: int,
        data_loader: torch.utils.data.DataLoader,
    ):
        batch_output_dict = {
            "loss": 0.0,
            "reward": 0.0,
            "actions": [],
            "origin_sentences": [],
            "augment_sentences": []
        }

        for episode_idx, episode in tqdm(enumerate(data_loader)):
            # feedforward and get loss
            if self.GPU >= 0:
                episode = move_to_device(episode, self.GPU)
            else:
                pass

            output_dict = self.train_model.forward(episode)

            # update batch dict
            batch_output_dict["loss"] += output_dict["loss"]
            batch_output_dict["reward"] += output_dict["ep_reward"]

            # batch updating
            if (episode_idx+1) % batch_size == 0:
                # Record
                print(self.record_step)
                print(output_dict["origin_sentence"])
                print(output_dict["augment_sentence"])
                batch_output_dict["origin_sentences"] += output_dict["origin_sentence"]
                batch_output_dict["augment_sentences"] += output_dict["augment_sentence"]
                batch_output_dict["actions"] += output_dict["actions"]

                if self.writer is None:
                    pass
                else:
                    self._record(self.record_step, batch_output_dict)

                self.record_step += 1

                # Optimize
                self.train_model.optimize(batch_output_dict["loss"] / batch_size)
                print(batch_output_dict["reward"] / batch_size)

                # Initialize
                batch_output_dict = {
                    "loss": 0.0,
                    "reward": 0.0,
                    "actions": [],
                    "origin_sentences": [],
                    "augment_sentences": []
                }

                # Save
                if self.is_save is True:
                    torch.save(
                        self.train_model.policy.state_dict(),
                        "model_record/reinforce_model_weights/policy" + str(self.record_step) + ".pkl"
                    )
                break

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
        train_model: torch.nn.Module,
        is_save: bool = False
    ):
        super(TextTrainer, self).__init__()
        self.train_model = train_model
        self.is_save = is_save

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

            output_dict = self.train_model.forward(batch)

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
        train_data_loader: torch.utils.data.DataLoader
    ):
        num_of_batch = 0
        total_loss = 0.0
        total_labels = []
        total_predicts = []

        for batch_idx, batch in enumerate(tqdm(train_data_loader)):
            if self.GPU >= 0:
                batch = move_to_device(batch, self.GPU)
            else:
                pass

            output_dict = self.train_model.forward(batch)

            # Optimize
            self.train_model.optimize(
                output_dict["classification_loss"],
                [self.train_model.optimizer]
            )

            num_of_batch += 1
            total_labels.append(batch["label"])
            total_predicts.append(output_dict["predicts"])
            total_loss += output_dict["classification_loss"].item()

        total_labels = torch.cat(total_labels, 0)
        total_predicts = torch.cat(total_predicts, 0)

        avg_loss = total_loss / num_of_batch
        avg_acc = torch.true_divide(torch.sum(total_labels == total_predicts), total_labels.shape[0])

        return avg_loss, avg_acc

    @overrides
    def fit(
        self,
        epochs: int,
        train_data_loader: torch.utils.data.DataLoader,
        valid_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader = None
    ):
        for epoch in tqdm(range(epochs)):
            # Do training
            self.train_model.train()
            train_avg_loss, train_avg_acc = self._fit_train(train_data_loader)

            # Do validation
            self.train_model.eval()
            with torch.no_grad():
                valid_avg_loss, valid_avg_acc = self._fit_valid(valid_data_loader)

            # Do testing
            self.train_model.eval()
            with torch.no_grad():
                test_avg_loss, test_avg_acc = self._fit_valid(test_data_loader)

            # LR schedulr
            if self.train_model.scheduler is not None:
                self.train_model.scheduler.step(valid_avg_loss)
            else:
                pass

            print("Epochs         : {}".format(epoch))
            print("Training Loss  : {:.5f}".format(train_avg_loss))
            print("Training Acc   : {:.5f}".format(train_avg_acc))
            print("Validation Loss: {:.5f}".format(valid_avg_loss))
            print("Validation Acc : {:.5f}".format(valid_avg_acc))
            print("Testing Loss   : {:.5f}".format(test_avg_loss))
            print("Testing Acc    : {:.5f}".format(test_avg_acc))
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
