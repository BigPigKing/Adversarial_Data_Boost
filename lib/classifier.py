import torch

from overrides import overrides
from allennlp.modules import FeedForward


class TextClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int
    ):
        super(TextClassifier, self).__init__()

        feedforward = FeedForward(
            input_dim=input_size,
            num_layers=2,
            hidden_dims=[300, 150],
            activations=torch.nn.ReLU(),
            dropout=0.3
        )

        final_linear = torch.nn.Linear(
            feedforward.get_output_dim(),
            output_size
        )

        self.classifiers = torch.nn.ModuleList(
            [feedforward, final_linear]
        )

    @overrides
    def forward(
        self,
        encode_X
    ):
        for classifier in self.classifiers:
            encode_X = classifier(encode_X)

        return encode_X

    def get_output_dim(
        self
    ) -> int:
        return self.classifiers[-1].get_output_dim()


def main():
    pass


if __name__ == '__main__':
    main()
