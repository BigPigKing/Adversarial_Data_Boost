import abc
import torch

from tqdm import tqdm
from overrides import overrides
from allennlp.nn.util import move_to_device


class Trainer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(
        self
    ):
        return NotImplemented


class ReinforceTrainer(Trainer):
    def __init__(
        self,
        train_model: torch.nn.Module
    ):
        super(ReinforceTrainer, self).__init__()
        self.train_model = train_model

    def _fit_epoch(
        self,
        data_loader: torch.utils.data.DataLoader
    ):
        for episode_idx, episode in tqdm(enumerate(data_loader)):
            episode = move_to_device(episode, 0)
            output_dict = self.train_model.forward(episode["tokens"])

            self.train_model.optimize(output_dict["loss"])

    def fit(
        self,
        epochs: int,
        data_loader: torch.utils.data.DataLoader
    ):
        for epoch in tqdm(range(epochs)):
            self.train_model.train()
            self._fit_epoch(data_loader)


class TextTrainer(Trainer):
    def __init__(
        self,
        train_model: torch.nn.Module
    ):
        super(TextTrainer, self).__init__()
        self.train_model = train_model

    def _fit_valid(
        self,
        valid_data_loader: torch.utils.data.DataLoader
    ):
        num_of_batch = 0
        total_loss = 0.0
        total_labels = []
        total_predicts = []

        for batch_idx, batch in enumerate(valid_data_loader):
            batch = move_to_device(batch, 0)
            output_dict = self.train_model.forward(batch["tokens"], batch["label"])

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
            batch = move_to_device(batch, 0)
            output_dict = self.train_model.forward(batch["tokens"], batch["label"])

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
            valid_avg_loss, valid_avg_acc = self._fit_valid(valid_data_loader)

            # Do testing
            self.train_model.eval()
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


def main():
    pass


if __name__ == '__main__':
    main()
